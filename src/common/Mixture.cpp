#include <iostream>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "Mixture.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
Mixture::Mixture(
    std::string model_name, double ds, std::map<std::string, double> bond_lengths, bool use_superposition)
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
        this->use_superposition = use_superposition;
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

    // temporary maps for the new polymer
    std::map<std::tuple<int, std::string>, std::map<std::tuple<int, std::string, int, int>, std::vector<std::tuple<int, int>>, std::greater<> >> unique_blocks_new_polymer;

    // find unique_blocks in new_polymer
    std::vector<PolymerChainBlock> blocks = pc.get_blocks();
    for(size_t b=0; b<blocks.size(); b++)
    {
        int v = blocks[b].v;
        int u = blocks[b].u;
        std::string dep_v = pc.get_dep(v, u);
        std::string dep_u = pc.get_dep(u, v);

        if (dep_v < dep_u){
            dep_v.swap(dep_u);
            std::swap(v,u);
        }

        auto key1 = std::make_tuple(distinct_polymers.size()-1, dep_v);
        auto key2 = std::make_tuple(blocks[b].n_segment, dep_u, 0, blocks[b].n_segment);
        unique_blocks_new_polymer[key1][key2].push_back(std::make_tuple(v,u));
    }

    if (use_superposition)
    {
        // find superposed branches in unique_blocks_new_polymer
        std::map<std::tuple<int, std::string>, std::map<std::tuple<int, std::string, int, int>, std::vector<std::tuple<int, int>>, std::greater<> >> superposed_blocks;
        for(auto& item : unique_blocks_new_polymer)
        {
            // remove branches that are already merged in the previous iteration
            std::vector<std::tuple<int, int>> total_v_u_list;
            for(auto& second_key : superposed_blocks[item.first]) // map <tuple, v_u_vector>
            {
                for(auto& superposition_v_u : second_key.second) 
                {
                    total_v_u_list.push_back(superposition_v_u);
                    // std::cout << "(v_u): " << std::get<0>(superposition_v_u) <<  ", " << std::get<1>(superposition_v_u) << std::endl;
                }
            }
            for(auto it = item.second.cbegin(); it != item.second.cend();) // map <tuple, v_u_vector>
            {
                bool removed = false;
                for(auto& v_u : total_v_u_list)
                {
                    if ( std::find(it->second.begin(), it->second.end(), v_u) != it->second.end())
                    {
                        it = item.second.erase(it);
                        removed = true;
                        break;
                    }
                }
                if (!removed)
                    ++it;
            }

            // add superposed branches
            for(auto& second_key : superposed_blocks[item.first]) 
            {
                unique_blocks_new_polymer[item.first][std::make_tuple(
                    std::get<0>(second_key.first),
                    std::get<1>(second_key.first),
                    std::get<2>(second_key.first),
                    std::get<3>(second_key.first))]
                    = second_key.second;
            }

            // find superposition
            std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>> dep_u_superposition_list;
            if (model_name == "continuous")
                dep_u_superposition_list = superpose_branches_continuous(item.second);
            else if (model_name == "discrete")
                dep_u_superposition_list = superpose_branches_discrete(item.second);

            // update list
            unique_blocks_new_polymer[item.first].clear();
            for(auto& dep_u_superposition : dep_u_superposition_list)
            {
                unique_blocks_new_polymer[item.first][std::make_tuple(
                    std::get<0>(dep_u_superposition),
                    std::get<1>(dep_u_superposition),
                    std::get<2>(dep_u_superposition),
                    std::get<3>(dep_u_superposition))]
                    = std::get<4>(dep_u_superposition);
            }

            // for each dep_u_superposition
            for(auto& dep_u_superposition : dep_u_superposition_list)
            {

                int n_segment_allocated = std::get<0>(dep_u_superposition);
                std::string dep_key = std::get<1>(dep_u_superposition);
                int n_segment_offset = std::get<2>(dep_u_superposition);
                int n_segment_original = std::get<3>(dep_u_superposition);

                // skip, if it is not superposed
                if ( dep_key[0] != '[' || n_segment_offset+n_segment_allocated != n_segment_original)
                    continue;

                // for each v_u 
                for(auto& v_u : std::get<4>(dep_u_superposition))
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

                            std::string new_u_key = "(" + dep_key
                                + std::to_string(n_segment_allocated);

                            for(auto& v_adj_node_dep : v_adj_nodes)
                            {
                                if (v_adj_node_dep != v && v_adj_node_dep != std::get<1>(v_u))
                                    new_u_key += pc.get_block(v_adj_node_dep,u).monomer_type + std::to_string(pc.get_block(v_adj_node_dep,u).n_segment);
                                
                            }
                            new_u_key += ")" + pc.get_block(v,u).monomer_type;

                            std::tuple<int, std::string, int, int> new_u_key_tuple = std::make_tuple(pc.get_block(v,u).n_segment, new_u_key, 0, pc.get_block(v,u).n_segment);

                            // std::cout << "new_u_key: " << pc.get_block(v,u).n_segment << ", " << new_u_key << std::endl;
                            superposed_blocks[key][new_u_key_tuple].push_back(std::make_tuple(v,u));
                        }
                    }
                }
            }
        }
    }

    // copy data
    for(const auto& v_item : unique_blocks_new_polymer)
    {
        for(const auto& u_item : v_item.second)
        {
            int polymer_id = std::get<0>(v_item.first);

            std::string key_v = std::get<1>(v_item.first);
            std::string key_u = std::get<1>(u_item.first);

            int n_segment_original = std::get<3>(u_item.first);
            int n_segment_allocated = std::get<0>(u_item.first);
            int n_segment_offset = std::get<2>(u_item.first);

            // add blocks
            auto key = std::make_tuple(polymer_id, key_v, key_u);

            unique_blocks[key].monomer_type = Mixture::key_to_species(key_v);
            unique_blocks[key].n_segment_allocated = n_segment_allocated;
            unique_blocks[key].n_segment_offset    = n_segment_offset;
            unique_blocks[key].n_segment_original  = n_segment_original;
            unique_blocks[key].v_u = u_item.second;

            // add branches
            add_unique_branch(unique_branches, key_v, n_segment_original);
            add_unique_branch(unique_branches, key_u, n_segment_allocated);
        }
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
bool Mixture::is_using_superposition() const
{
    return use_superposition;
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
    for(size_t i=0; i<adjacent_nodes[in_node].size(); i++)
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
        std::sort(edge_text.begin(), edge_text.end());
        text += "(";
        for(size_t i=0; i<edge_text.size(); i++)
            text += edge_text[i];
        text += ")";
    }

    // update unique_sub_branches
    text += blocks[edge_to_array[std::make_pair(in_node, out_node)]].monomer_type;

    return std::make_pair(text, blocks[edge_to_array[std::make_pair(in_node, out_node)]].n_segment);
}
void Mixture::add_unique_branch(std::map<std::string, UniqueEdge, CompareBranchKey>& unique_branches, std::string new_key, int new_n_segment)
{
    if (unique_branches.find(new_key) == unique_branches.end())
    {
        unique_branches[new_key].deps = Mixture::key_to_deps(new_key);
        unique_branches[new_key].monomer_type = Mixture::key_to_species(new_key);
        unique_branches[new_key].max_n_segment = new_n_segment;
        unique_branches[new_key].height = Mixture::key_to_height(new_key);
    }
    else
    {
        if (unique_branches[new_key].max_n_segment < new_n_segment)
            unique_branches[new_key].max_n_segment = new_n_segment;
    }
}
std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>>
    Mixture::superpose_branches_continuous(std::map<std::tuple<int, std::string, int, int>, std::vector<std::tuple<int, int>>, std::greater<>> u_map)
{
    // Example)
    // 0, B:
    //   6, 0, 6, (C)B, 1,       6
    //   6, 0, 6, (D)B, 3,       6
    //   6, 0, 6, (E)B, 2,       6
    //   4, 0, 4, (F)B, 1,       4
    //
    //      ↓   Superposition
    //  
    //   6, 0, 0, (C)B, 1,                      6
    //   6, 0, 0, (D)B, 3,                      6
    //   6, 0, 0, (E)B, 2,                      6
    //   4, 0, 4, (F)B, 1,                      4
    //   6, 0, 6, [(C)B0:1,(D)B0:3,(E)B0:2]B,   6
    //
    //      ↓   Forward up to the second largest n_segment
    //  
    //   6, 0, 0, (C)B, 1,                               6
    //   6, 0, 0, (D)B, 3,                               6 
    //   6, 0, 0, (E)B, 2,                               6 
    //   4, 0, 4, (F)B, 1,                               4
    //   6, 0, 2, [(C)B0:1,(D)B0:3,(E)B0:2]B,            6
    //   6, 2, 4, ([(C)B0:1,(D)B0:3,(E)B0:2]B2)B,        6
    //
    //      ↓   Superposition
    //  
    //   6, 0, 0, (C)B, 1,                                  6
    //   6, 0, 0, (D)B, 3,                                  6 
    //   6, 0, 0, (E)B, 2,                                  6 
    //   4, 0, 0, (F)B, 1,                                  4
    //   6, 0, 2, [(C)B0:1,(D)B0:3,(E)B0:2]B,               6
    //   6, 2, 0, ([(C)B0:1,(D)B0:3,(E)B0:2]B2)B,           6
    //   6, 2, 4, [([(C)B0:1,(D)B0:3,(E)B0:2]B2)B0,(F)B0],  4

    std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>> remaining_keys;
    std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>> dep_u_superposition_list;
    std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>> dep_u_superposition_total;

    // Because of our SimpsonQuadrature implementation, whose weights of odd number n_segments and even number n_segments are slightly different,
    // superpositions for blocks of odd number and of even number are separately performed.

    // for even number
    for(const auto& item : u_map)
    {
        if (std::get<0>(item.first) % 2 == 0)
        {
            remaining_keys.push_back(std::make_tuple(
                std::get<0>(item.first), std::get<1>(item.first),
                std::get<2>(item.first), std::get<3>(item.first), item.second));
        }
    }
    dep_u_superposition_total = superpose_branches_common(remaining_keys, 0);

    // for odd number
    remaining_keys.clear();
    for(const auto& item : u_map)
    {
        if (std::get<0>(item.first) % 2 == 1)
        {
            remaining_keys.push_back(std::make_tuple(
                std::get<0>(item.first), std::get<1>(item.first),
                std::get<2>(item.first), std::get<3>(item.first), item.second));
        }
    }
    dep_u_superposition_list = superpose_branches_common(remaining_keys, 0);

    // merge vectors
    dep_u_superposition_total.insert(std::end(dep_u_superposition_total), std::begin(dep_u_superposition_list), std::end(dep_u_superposition_list));

    return dep_u_superposition_total;

}
std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>>
    Mixture::superpose_branches_discrete(std::map<std::tuple<int, std::string, int, int>, std::vector<std::tuple<int, int>>, std::greater<>> u_map)
{

    // Example)
    // 0, B:
    //   7, 0, 7, (C)B, 1,
    //   7, 0, 7, (D)B, 3,
    //   7, 0, 7, (E)B, 2,
    //   4, 0, 4, (F)B, 1,
    //
    //      ↓   Superposition
    //  
    //   7, 0, 1, (C)B, 1,
    //   7, 0, 1, (D)B, 3,
    //   7, 0, 1, (E)B, 2,
    //   7, 1, 6, [(C)B1:1,(D)B1:3,(E)B1:2]B
    //   4, 0, 4, (F)B, 1
    //
    //      ↓  Forward up to the second largest n_segment
    //  
    //   7, 0, 1, (C)B, 1,
    //   7, 0, 1, (D)B, 3,
    //   7, 0, 1, (E)B, 2,
    //   7, 1, 1, [(C)B1:1,(D)B1:3,(E)B1:2]B
    //   7, 3, 4, ([(C)B1:1,(D)B1:3,(E)B1:2]B2)B
    //   4, 0, 4, (F)B, 1
    //
    //      ↓   Superposition
    //  
    //   7, 0, 1, (C)B, 1,
    //   7, 0, 1, (D)B, 3,
    //   7, 0, 1, (E)B, 2,
    //   7, 1, 1, [(C)B1:1,(D)B1:3,(E)B1:2]B
    //   7, 3, 1, ([(C)B1:1,(D)B1:3,(E)B1:2]B2)B
    //   4, 0, 1, (F)B, 1
    //   7, 4, 3, [([(C)B1:1,(D)B1:3,(E)B1:2]B2)B1,:(F)B1:1]B

    std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>> remaining_keys;

    for(const auto& item : u_map)
        remaining_keys.push_back(std::make_tuple(
            std::get<0>(item.first), std::get<1>(item.first),
            std::get<2>(item.first), std::get<3>(item.first), item.second));

    return superpose_branches_common(remaining_keys, 1);
}


std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>>
    Mixture::superpose_branches_common(std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>> remaining_keys, int minimum_n_segment)
{
    int current_n_segment = std::get<0>(remaining_keys[0]);
    int n_segment_allocated;
    int n_segment_offset;
    int n_segment_original;

    std::string dep_u_superposition;
    std::vector<std::tuple<int ,int>> v_u_total;

    // tuple <n_segment_allocated, key, n_segment_offset, n_segment_original, v_u_list>
    std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>> dep_u_superposition_list;
    std::vector<std::tuple<int, std::string, int, int, std::vector<std::tuple<int ,int>>>> level_superposition_list;

    // std::cout << "---------map------------" << std::endl;
    // for(const auto& item : remaining_keys)
    // {
    //     std::cout << std::get<0>(item) << ", " <<
    //                  std::get<1>(item) << ", " <<
    //                  std::get<2>(item) << ", " <<
    //                  std::get<3>(item) << ", ";
    //     for(const auto& v_u : std::get<4>(item))
    //     {
    //         std::cout << "("
    //         + std::to_string(std::get<0>(v_u)) + ","
    //         + std::to_string(std::get<1>(v_u)) + ")" + ", ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "-----------------------" << std::endl;

    // int count = 0;
    if (!remaining_keys.empty())
        current_n_segment = std::get<0>(remaining_keys[0]);
    while(!remaining_keys.empty())
    {
        // count ++;
        // if (count == 10)
        //     break;
        // std::cout << "remaining_keys" << std::endl;
        // for(size_t i=0; i<remaining_keys.size(); i++)
        // {
        //     std::cout << std::get<1>(remaining_keys[i]) << ", "  
        //               << std::get<0>(remaining_keys[i]) << ", " 
        //               << std::get<2>(remaining_keys[i]) << ", "
        //               << std::get<3>(remaining_keys[i]) << ", " << std::endl;
        // }
        // std::cout << "-------------" << std::endl;

        std::set<int, std::greater<int>> n_segment_set;
        for(size_t i=0; i<remaining_keys.size(); i++)
        {
            if (std::get<0>(remaining_keys[i]) <= 1)
            {
                dep_u_superposition_list.push_back(remaining_keys[i]);
                remaining_keys.erase(std::remove(remaining_keys.begin(), remaining_keys.end(), remaining_keys[i]), remaining_keys.end());
                continue;
            }
            if (current_n_segment == std::get<0>(remaining_keys[i]))
                level_superposition_list.push_back(remaining_keys[i]);
            n_segment_set.insert(std::get<0>(remaining_keys[i]));
        }

        // if it empty, decrease current_n_segment.
        if (level_superposition_list.empty())
            current_n_segment = std::get<0>(remaining_keys[0]);
        else
        {
            // for(size_t i=0; i<level_superposition_list.size(); i++)
            //     std::cout << std::get<0>(level_superposition_list[i]) << ", " << std::get<1>(level_superposition_list[i]) << std::endl;

            v_u_total.clear();
            // if there is only one element
            if (level_superposition_list.size() == 1)
            {
                //  No the second largest key
                if (n_segment_set.size() == 1)
                {
                    // add to vectors
                    dep_u_superposition_list.push_back(level_superposition_list[0]);
                }
                // Forward up to the second largest n_segment
                else
                {
                    int second_largest_n_segments = *std::next(n_segment_set.begin(), 1);

                    std::string dep_key = std::get<1>(level_superposition_list[0]);
                    n_segment_offset = std::get<2>(level_superposition_list[0]);
                    n_segment_original = std::get<3>(level_superposition_list[0]);
                    n_segment_allocated = n_segment_original-second_largest_n_segments;

                    std::vector<std::tuple<int ,int>> dep_v_u = std::get<4>(level_superposition_list[0]);
                    v_u_total.insert(v_u_total.end(), dep_v_u.begin(), dep_v_u.end());

                    dep_u_superposition = "(" + dep_key + std::to_string(n_segment_allocated);
                    if (dep_key.find(')') == std::string::npos)
                        dep_u_superposition += ":" + std::to_string(dep_v_u.size());
                    dep_u_superposition += ")" + Mixture::key_to_species(dep_key);
                    
                    // add to vector
                    dep_u_superposition_list.push_back(std::make_tuple(
                        n_segment_allocated, dep_key, n_segment_offset, n_segment_original, dep_v_u));
                    remaining_keys.insert(remaining_keys.begin(),std::make_tuple(
                        second_largest_n_segments, dep_u_superposition,
                        n_segment_allocated, n_segment_original, dep_v_u));
                }
            }
            // Superposition
            else
            {
                // sort level_superposition_list with second element
                std::sort(level_superposition_list.begin(), level_superposition_list.end(),
                    [](auto const &t1, auto const &t2)
                        {
                            return Mixture::key_to_height(std::get<1>(t1)) > Mixture::key_to_height(std::get<1>(t2));
                        }
                );

                // add one by one
                if (n_segment_set.size() == 1)
                    n_segment_allocated = minimum_n_segment;
                else
                    n_segment_allocated = std::get<0>(level_superposition_list[0])-*std::next(n_segment_set.begin(), 1);

                std::string dep_key = std::get<1>(level_superposition_list[0]);
                n_segment_offset = std::get<2>(level_superposition_list[0]);
                n_segment_original = std::get<3>(level_superposition_list[0]);
                std::vector<std::tuple<int ,int>> dep_v_u = std::get<4>(level_superposition_list[0]);

                int n_segment_offset_max = n_segment_offset;
                int n_segment_original_max = n_segment_original;

                v_u_total.insert(v_u_total.end(), dep_v_u.begin(), dep_v_u.end());

                dep_u_superposition = "[" + dep_key + std::to_string(n_segment_allocated);
                if (dep_key.find('[') == std::string::npos)
                    dep_u_superposition += ":" + std::to_string(dep_v_u.size());

                // add to vector
                dep_u_superposition_list.push_back(std::make_tuple(n_segment_allocated, dep_key, n_segment_offset, n_segment_original, dep_v_u));

                for(size_t i=1; i<level_superposition_list.size(); i++)
                {
                    dep_key = std::get<1>(level_superposition_list[i]);
                    n_segment_offset = std::get<2>(level_superposition_list[i]);
                    n_segment_original = std::get<3>(level_superposition_list[i]);
                    dep_v_u = std::get<4>(level_superposition_list[i]);

                    n_segment_offset_max = std::max(n_segment_offset_max, n_segment_offset);
                    n_segment_original_max = std::max(n_segment_original_max, n_segment_original);

                    v_u_total.insert(v_u_total.end(), dep_v_u.begin(), dep_v_u.end());

                    dep_u_superposition += "," + dep_key + std::to_string(n_segment_allocated);
                    if (dep_key.find('[') == std::string::npos)
                        dep_u_superposition += ":" + std::to_string(dep_v_u.size());

                    // add to vector
                    dep_u_superposition_list.push_back(std::make_tuple(n_segment_allocated, dep_key, n_segment_offset, n_segment_original, dep_v_u));
                }
                dep_u_superposition += "]" + Mixture::key_to_species(dep_key);

                // add one by one
                n_segment_offset_max += n_segment_allocated;
                n_segment_allocated = std::get<0>(level_superposition_list[0])-n_segment_allocated;

                // add to vector
                remaining_keys.insert(remaining_keys.begin(), std::make_tuple(n_segment_allocated, dep_u_superposition, n_segment_offset_max, n_segment_original_max, v_u_total));
            }
            // erase elements
            for(size_t i=0; i<level_superposition_list.size(); i++)
                remaining_keys.erase(std::remove(remaining_keys.begin(), remaining_keys.end(), level_superposition_list[i]), remaining_keys.end());
            level_superposition_list.clear();
        }
    }
    return dep_u_superposition_list;
}

int Mixture::get_unique_n_branches() const
{
    return unique_branches.size();
}
std::vector<std::tuple<std::string, int, int>> Mixture::key_to_deps(std::string key)
{
    std::vector<std::tuple<std::string, int, int>> sub_deps;
    int sub_n_segment;
    std::string sub_key;
    int sub_n_repeated;

    bool is_reading_key = true;
    bool is_reading_n_segment = false;
    bool is_reading_n_repeated = false;

    int key_start = 1;
    int brace_count = 0;

    for(size_t i=0; i<key.size();i++)
    {
        // it was reading key and found a digit
        if( isdigit(key[i]) && is_reading_key && brace_count == 1 )
        {
            // std::cout << "key_to_deps1" << std::endl;
            sub_key = key.substr(key_start, i-key_start);
            // std::cout << sub_key << "= " << key_start << ", " << i  << std::endl;

            is_reading_key = false;
            is_reading_n_segment = true;

            key_start = i;
        }
        // it was reading n_segment and found a ':'
        else if( key[i]==':' && is_reading_n_segment && brace_count == 1 )
        {
            // std::cout << "key_to_deps2" << std::endl;
            sub_n_segment = std::stoi(key.substr(key_start, i-key_start));
            // std::cout << sub_key << "= " << key_start << ", " << i  << ", " << key.substr(key_start, i-key_start) << std::endl;

            is_reading_n_segment = false;
            is_reading_n_repeated = true;

            key_start = i+1;
        }
        // it was reading n_segment and found a comma
        else if( key[i]==',' && is_reading_n_segment && brace_count == 1 )
        {
            // std::cout << "key_to_deps3" << std::endl;
            sub_n_segment = std::stoi(key.substr(key_start, i-key_start));
            // std::cout << sub_key << "= " << key_start << ", " << i  << ", " << key.substr(key_start, i-key_start) << std::endl;
            sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, 1));

            is_reading_n_segment = false;
            is_reading_key = true;

            key_start = i+1;
        }
        // it was reading n_repeated and found a comma
        else if( key[i]==',' && is_reading_n_repeated && brace_count == 1 )
        {
            // std::cout << "key_to_deps4" << std::endl;
            sub_n_repeated = std::stoi(key.substr(key_start, i-key_start));
            // std::cout << sub_key << "= " << key_start << ", " << i  << ", " << key.substr(key_start, i-key_start) << std::endl;
            sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, sub_n_repeated));

            is_reading_n_repeated = false;
            is_reading_key = true;

            key_start = i+1;
        }
        // it was reading n_repeated and found a non-digt
        else if( !isdigit(key[i]) && is_reading_n_repeated && brace_count == 1)
        {
            // std::cout << "key_to_deps5" << std::endl;
            sub_n_repeated = std::stoi(key.substr(key_start, i-key_start));
            // std::cout << sub_key << "= " << key_start << ", " << i  << ", " << key.substr(key_start, i-key_start) << std::endl;
            sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, sub_n_repeated));

            is_reading_n_repeated = false;
            is_reading_key = true;

            key_start = i;
        }
        // it was reading n_segment and found a non-digt
        else if( !isdigit(key[i]) && is_reading_n_segment && brace_count == 1)
        {
            // std::cout << "key_to_deps6" << std::endl;
            sub_n_segment = std::stoi(key.substr(key_start, i-key_start));
            // std::cout << sub_key << "= " << key_start << ", " << i  << ", " << key.substr(key_start, i-key_start) << std::endl;
            sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, 1));

            is_reading_n_segment = false;
            is_reading_key = true;

            key_start = i;
        }
        if(key[i] == '(' || key[i] == '[')
            brace_count++;
        else if(key[i] == ')' || key[i] == ']')
            brace_count--;
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
        for(size_t i=0; i<key.size();i++)
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
    for(size_t i=0; i<key.size();i++)
    {
        if (key[i] == '[' || key[i] == '(')
            height_count++;
        else
            break;
    }
    return height_count;
}
std::map<std::string, UniqueEdge, CompareBranchKey>& Mixture::get_unique_branches()
{
    return unique_branches;
}
UniqueEdge& Mixture::get_unique_branch(std::string key)
{
    if (unique_branches.find(key) == unique_branches.end())
        throw_with_line_number("There is no such key (" + key + ").");
    return unique_branches[key];
}
std::map<std::tuple<int, std::string, std::string>, UniqueBlock>& Mixture::get_unique_blocks()
{
    return unique_blocks;
}
UniqueBlock& Mixture::get_unique_block(std::tuple<int, std::string, std::string> key)
{
    // assert(("There is no such key (" + std::to_string(std::get<0>(key)) + ", " + 
    //     std::get<1>(key) + ", " + std::get<2>(key) + ", " + std::to_string(std::get<3>(key)) + ").", unique_blocks.count(key) != 0));
    assert(unique_blocks.count(key) != 0 && "There is no such key.");
    return unique_blocks[key];
}
void Mixture::display_unique_blocks() const
{
    // print Unique Blocks
    if (use_superposition)
    {
        std::cout << "--------- Unique Blocks (Superposed) ---------" << std::endl;
        std::cout << "Polymer id, key1:\n\tn_segment (original, offset, allocated), key2, (v, u)" << std::endl;
    }
    else
    {
        std::cout << "--------- Unique Blocks ---------" << std::endl;
        std::cout << "Polymer id, key1:\n\tn_segment, key2, (v, u)" << std::endl;
    }

    std::tuple<int, std::string> v_string = std::make_tuple(-1, "");

    for(const auto& item : unique_blocks)
    {
        const auto& v_key = item.first;

        if (v_string != std::make_tuple(std::get<0>(v_key),std::get<1>(v_key)))
        {
            std::cout <<
                std::to_string(std::get<0>(v_key)) + ", " +
                std::get<1>(v_key) + ":" << std::endl;

            v_string = std::make_tuple(std::get<0>(v_key),std::get<1>(v_key));
        }

        if (use_superposition)
            std::cout << "\t(" + std::to_string(item.second.n_segment_original) + ", "+ std::to_string(item.second.n_segment_offset) + ", " + std::to_string(item.second.n_segment_allocated) + "), " + std::get<2>(v_key) +  ", ";
        else
            std::cout << "\t" + std::to_string(item.second.n_segment_allocated) + ", " + std::get<2>(v_key) +  ", ";

        for(const auto& v_u : item.second.v_u)
        {
            std::cout << "("
            + std::to_string(std::get<0>(v_u)) + ","
            + std::to_string(std::get<1>(v_u)) + ")" + ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}
void Mixture::display_unique_branches() const
{
    // print unique sub branches
    std::vector<std::tuple<std::string, int, int>> sub_deps;
    int total_segments = 0;
    if (use_superposition)
        std::cout << "--------- Unique Branches (Superposed) ---------" << std::endl;
    else
        std::cout << "--------- Unique Branches ---------" << std::endl;
    for(const auto& item : unique_branches)
    {
        total_segments += item.second.max_n_segment;

        std::cout << item.first;
        std::cout << ":\n\tmax_n_segment: " << item.second.max_n_segment;
        std::cout << ", height: " << item.second.height;
        std::cout << ",\n\tsub_deps:{ ";
        sub_deps = key_to_deps(item.first);
        for(size_t i=0; i<sub_deps.size(); i++)
        {
            std::cout << std::get<0>(sub_deps[i]) << ":" << std::get<1>(sub_deps[i]) << ", " ;
        }
        std::cout << "}" << std::endl;
    }
    std::cout << "The number of total segments to be computed: " << total_segments << std::endl;
    std::cout << "------------------------------------" << std::endl;
}