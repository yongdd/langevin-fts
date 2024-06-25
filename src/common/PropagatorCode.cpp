#include <iostream>
#include <cstdio>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "Molecules.h"
#include "PropagatorCode.h"
#include "Exception.h"

// //----------------- Constructor ----------------------------
// PropagatorCode::PropagatorCode()
// {
// }

std::vector<std::tuple<int, int, std::string>> PropagatorCode::generate_codes(
    Polymer& pc, std::map<int, std::string>& chain_end_to_q_init)
{
    std::vector<std::tuple<int, int, std::string>> propagator_codes;    
    std::string propagator_code; // code, n_segment

    // Generate propagator code for each block and each direction
    std::map<std::pair<int, int>, std::string> memory;
    for (size_t b=0; b<pc.get_blocks().size(); b++)
    {
        int v = pc.get_blocks()[b].v;
        int u = pc.get_blocks()[b].u;
        propagator_code = PropagatorCode::generate_edge_code(
            memory,
            pc.get_blocks(),
            pc.get_adjacent_nodes(),
            pc.get_block_indexes(),
            chain_end_to_q_init,
            v, u);
        
        propagator_codes.push_back(std::make_tuple(v, u, propagator_code));

        propagator_code = PropagatorCode::generate_edge_code(
            memory,
            pc.get_blocks(),
            pc.get_adjacent_nodes(),
            pc.get_block_indexes(),
            chain_end_to_q_init,
            u, v);
            
        propagator_codes.push_back(std::make_tuple(u, v, propagator_code));
    }
    return propagator_codes;
}

std::string PropagatorCode::generate_edge_code(
    std::map<std::pair<int, int>, std::string>& memory,
    std::vector<Block>& blocks,
    std::map<int, std::vector<int>>& adjacent_nodes,
    std::map<std::pair<int, int>, int>& edge_to_block_index,
    std::map<int, std::string>& chain_end_to_q_init,
    int in_node, int out_node)
{
    std::vector<std::string> queue_sub_codes;
    std::string sub_code;

    // If it is already computed
    if (memory.find(std::make_pair(in_node, out_node)) != memory.end())
        return memory[std::make_pair(in_node, out_node)];

    // Explore neighbor nodes
    //std::cout << "[" + std::to_string(in_node) + ", " +  std::to_string(out_node) + "]:";
    for(size_t i=0; i<adjacent_nodes[in_node].size(); i++)
    {
        if (adjacent_nodes[in_node][i] != out_node)
        {
            //std::cout << "(" << in_node << ", " << adjacent_nodes[in_node][i] << ")";
            auto v_u_pair = std::make_pair(adjacent_nodes[in_node][i], in_node);
            if (memory.find(v_u_pair) != memory.end())
                sub_code = memory[v_u_pair];
            else
            {
                sub_code = generate_edge_code(
                    memory, blocks, adjacent_nodes,
                    edge_to_block_index, chain_end_to_q_init,
                    adjacent_nodes[in_node][i], in_node);
                memory[v_u_pair] = sub_code;
            }
            queue_sub_codes.push_back(sub_code);
        }
    }

    // Merge code of child propagators
    std::string code = "";
    if(queue_sub_codes.size() != 0)
    {
        std::sort(queue_sub_codes.begin(), queue_sub_codes.end());
        code += "(";
        for(size_t i=0; i<queue_sub_codes.size(); i++)
            code += queue_sub_codes[i];
        code += ")";
    }
    // If in_node exists in chain_end_to_q_init
    else if (chain_end_to_q_init.find(in_node) != chain_end_to_q_init.end())
    {
        code = "{" + chain_end_to_q_init[in_node] + "}";
    }

    // Add monomer_type at the end of code
    code += blocks[edge_to_block_index[std::make_pair(in_node, out_node)]].monomer_type;
    code += std::to_string(blocks[edge_to_block_index[std::make_pair(in_node, out_node)]].n_segment);

    // Save the result in memory
    memory[std::make_pair(in_node, out_node)] = code;

    return code;
}

std::string PropagatorCode::get_key_from_code(std::string code)
{
    int pos;
    for(int i=code.size()-1; i>=0;i--)
    {
        if(isalpha(code[i]))
        {
            pos = i+1;
            break;
        }
    }
    //std::cout<<"code.substr(0, pos); " << code << "  " << pos << " " << code.substr(0, pos) << std::endl;
    return code.substr(0, pos);
}

std::vector<std::tuple<std::string, int, int>> PropagatorCode::get_deps_from_key(std::string key)
{
    // sub_key, sub_n_segment, sub_n_repeated
    std::vector<std::tuple<std::string, int, int>> sub_deps;
    std::string sub_key;
    int sub_n_segment;
    int sub_n_repeated;

    int pos_start = 1; // start from 1 to remove open parenthesis
    std::stack<char> stack_brace;
    int state = 1;

    for(size_t i=0; i<key.size();i++)
    {
        if(stack_brace.size() == 1)
        {
            // It was reading key and have found a digit
            if(state==1 && isdigit(key[i]))
            {
                // Remove a comma
                if (key[pos_start] == ',')
                    pos_start += 1;

                sub_key = key.substr(pos_start, i-pos_start);
                // printf("%5d,%5d\n", pos_start, i);
                // std::cout << key << "," << key.substr(pos_start, i-pos_start) << std::endl;

                state = 2;
                pos_start = i;
            }
            // It was reading n_segment and have found a ':'
            else if(state==2 && key[i]==':')
            {
                sub_n_segment = std::stoi(key.substr(pos_start, i-pos_start));
                // printf("%5d,%5d=", pos_start, i);
                // std::cout << key.substr(pos_start, i-pos_start) << std::endl;

                state = 3;
                pos_start = i+1;
            }
            // It was reading n_segment and have found a non-digit
            else if(state==2 && !isdigit(key[i]))
            {
                sub_n_segment = std::stoi(key.substr(pos_start, i-pos_start));
                // printf("%5d,%5d=", pos_start, i);
                // std::cout << key.substr(pos_start, i-pos_start) << std::endl;

                sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, 1));

                state = 1;
                pos_start = i;
            }
            // It was reading n_repeated and have found a non-digit
            else if(state==3 && !isdigit(key[i]))
            {
                sub_n_repeated = std::stoi(key.substr(pos_start, i-pos_start));
                // printf("%5d,%5d=", pos_start, i);
                // std::cout << key.substr(pos_start, i-pos_start) << std::endl;
                
                sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, sub_n_repeated));
                state = 1;
                pos_start = i;
            }
        }
        if(key[i] == '(' || key[i] == '[')
        {
            stack_brace.push(key[i]);
        }
        else if(key[i] == ')' || key[i] == ']')
        {
            if((stack_brace.top() == '(' && key[i] == ')') ||
               (stack_brace.top() == '[' && key[i] == ']'))
            {
                stack_brace.pop();
            }
            else
            {
                throw_with_line_number("Failed to match parentheses or braces.");
            }
        }
    }
    return sub_deps;
}

std::string PropagatorCode::remove_monomer_type_from_key(std::string key)
{
    if (key[0] != '[' && key[0] != '(' && key[0] != '{')
    {
        return "";
    }
    else
    {
        int brace_depth = 0;
        int species_idx = 0;
        for(size_t i=0; i<key.size();i++)
        {
            if (key[i] == '[' || key[i] == '(' || key[i] == '{')
            {
                brace_depth++;
            }
            else if (key[i] == ']' || key[i] == ')' || key[i] == '}')
            {
                brace_depth--;
                if (brace_depth == 0)
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

std::string PropagatorCode::get_monomer_type_from_key(std::string key)
{
    int pos_start = 0;
    for(int i=key.size()-1; i>=0;i--)
    {
        //std::cout << key[i] << std::endl;
        if(key[i] == ')' || key[i] == ']' || key[i] == '}')
        {
            pos_start=i+1;
            break;
        }
    }
    //std::cout << key.substr(pos_start, key.size()-pos_start) << std::endl;
    return key.substr(pos_start, key.size()-pos_start);
}
std::string PropagatorCode::get_q_input_idx_from_key(std::string key)
{
    if (key[0] != '{')
        throw_with_line_number("There is no related initial condition in key (" + key + ").");

    int pos_start = 0;
    for(int i=key.size()-1; i>=0;i--)
    {
        if(key[i] == '}')
        {
            pos_start=i;
            break;
        }
    }
    // std::cout << key.substr(1, pos_start-1) << std::endl;
    return key.substr(1, pos_start-1);
}
int PropagatorCode::get_height_from_key(std::string key)
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