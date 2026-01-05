/**
 * @file PropagatorCode.cpp
 * @brief Implementation of propagator key generation and parsing.
 *
 * Provides static utility functions for generating unique string codes
 * that identify propagator computations. These codes encode the polymer
 * topology and enable detection of equivalent propagators across different
 * chains in a mixture.
 *
 * **Code Format:**
 *
 * Propagator keys encode the path from chain ends to a given edge:
 * - Simple chain end: "A10" (monomer type A, 10 segments)
 * - Junction: "(A10B5)C8" (two branches meeting, then C block)
 * - Aggregated: "[A10:2,B5]C8" (two identical A branches aggregated)
 * - Custom q_init: "{init_key}A10" (custom initial condition)
 *
 * **Key Components:**
 *
 * - Monomer type: Single letter or string (e.g., "A", "B_long")
 * - Segment count: Integer following monomer type
 * - Parentheses (): Junction point merging multiple branches
 * - Brackets []: Aggregated computation of identical branches
 * - Braces {}: Custom initial condition reference
 * - Colon: Repetition count (e.g., "A10:3" = three identical A10 branches)
 *
 * @see PropagatorComputationOptimizer for optimization using these codes
 * @see Polymer for chain topology
 */

#include <iostream>
#include <cstdio>
#include <cctype>
#include <cmath>
#include <numbers>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "Molecules.h"
#include "PropagatorCode.h"
#include "ContourLengthMapping.h"
#include "Exception.h"
#include "ValidationUtils.h"

/**
 * @brief Generate propagator codes for all edges in a polymer.
 *
 * Traverses the polymer graph and generates unique string codes for
 * each directed edge (v→u). Uses memoization to avoid recomputing
 * codes for shared sub-paths.
 *
 * @param pc                  Polymer graph to process
 * @param chain_end_to_q_init Map of chain ends to custom initial conditions
 *
 * @return Vector of (v, u, code) tuples for each directed edge
 *
 * @see generate_edge_code for recursive code generation
 */
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

/**
 * @brief Recursively generate code for a single directed edge.
 *
 * Builds the propagator code by recursively traversing from chain ends
 * toward the target edge. At junctions, child codes are sorted and
 * merged with parentheses notation.
 *
 * **Algorithm:**
 *
 * 1. Check memoization cache for existing result
 * 2. Recursively generate codes for all incoming edges (except outgoing)
 * 3. Sort and merge child codes with repetition counts
 * 4. Append current block's monomer type and segment count
 *
 * @param memory             Memoization cache (modified in place)
 * @param blocks             Polymer blocks
 * @param adjacent_nodes     Adjacency list for polymer graph
 * @param edge_to_block_index Map from (v,u) to block index
 * @param chain_end_to_q_init Custom initial conditions
 * @param in_node            Source node of edge
 * @param out_node           Target node of edge
 *
 * @return String code for the edge (in_node → out_node)
 */
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
    if (validation::contains(memory, std::make_pair(in_node, out_node)))
        return memory[std::make_pair(in_node, out_node)];

    // Explore neighbor nodes
    //std::cout << "[" + std::to_string(in_node) + ", " +  std::to_string(out_node) + "]:";
    for(size_t i=0; i<adjacent_nodes[in_node].size(); i++)
    {
        if (adjacent_nodes[in_node][i] != out_node)
        {
            //std::cout << "(" << in_node << ", " << adjacent_nodes[in_node][i] << ")";
            auto v_u_pair = std::make_pair(adjacent_nodes[in_node][i], in_node);
            if (validation::contains(memory, v_u_pair))
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
        // Count each sub_code
        std::vector<std::string> unique_sub_codes;
        std::vector<int> counts;
        for(size_t i=0; i<queue_sub_codes.size(); i++)
        {
            if (i==0 || queue_sub_codes[i] != queue_sub_codes[i-1])
            {
                unique_sub_codes.push_back(queue_sub_codes[i]);
                counts.push_back(1);
            }
            else
            {
                counts[counts.size()-1] += 1;
            }
        }
        // Merge codes
        code += "(";
        for(size_t i=0; i<unique_sub_codes.size(); i++)
        {
            code += unique_sub_codes[i];
            if (counts[i] > 1)
            {
                code += ":" + std::to_string(counts[i]);
                // if (i != unique_sub_codes.size()-1)
                //     code += ",";
            }
        }
        code += ")";

        // code += "(";
        // for(size_t i=0; i<queue_sub_codes.size(); i++)
        //     code += queue_sub_codes[i];
        // code += ")";
    }
    // If in_node exists in chain_end_to_q_init
    else if (validation::contains(chain_end_to_q_init, in_node))
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

/**
 * @brief Generate propagator codes using length index mapping.
 *
 * Traverses the polymer graph and generates unique string codes for
 * each directed edge (v→u), using ContourLengthMapping to convert
 * contour_length to integer indices.
 *
 * @param pc                  Polymer graph to process
 * @param chain_end_to_q_init Map of chain ends to custom initial conditions
 * @param mapping             Contour length mapping (must be finalized)
 *
 * @return Vector of (v, u, code) tuples for each directed edge
 */
std::vector<std::tuple<int, int, std::string>> PropagatorCode::generate_codes_with_mapping(
    Polymer& pc, std::map<int, std::string>& chain_end_to_q_init,
    const ContourLengthMapping& mapping)
{
    std::vector<std::tuple<int, int, std::string>> propagator_codes;
    std::string propagator_code;

    // Generate propagator code for each block and each direction
    std::map<std::pair<int, int>, std::string> memory;
    for (size_t b=0; b<pc.get_blocks().size(); b++)
    {
        int v = pc.get_blocks()[b].v;
        int u = pc.get_blocks()[b].u;
        propagator_code = PropagatorCode::generate_edge_code_with_mapping(
            memory,
            pc.get_blocks(),
            pc.get_adjacent_nodes(),
            pc.get_block_indexes(),
            chain_end_to_q_init,
            mapping,
            v, u);

        propagator_codes.push_back(std::make_tuple(v, u, propagator_code));

        propagator_code = PropagatorCode::generate_edge_code_with_mapping(
            memory,
            pc.get_blocks(),
            pc.get_adjacent_nodes(),
            pc.get_block_indexes(),
            chain_end_to_q_init,
            mapping,
            u, v);

        propagator_codes.push_back(std::make_tuple(u, v, propagator_code));
    }
    return propagator_codes;
}

/**
 * @brief Recursively generate code using length index.
 *
 * Similar to generate_edge_code but uses ContourLengthMapping
 * to convert contour_length to integer index.
 *
 * @param memory             Memoization cache
 * @param blocks             Polymer blocks
 * @param adjacent_nodes     Adjacency list
 * @param edge_to_block_index Edge to block index mapping
 * @param chain_end_to_q_init Custom initial conditions
 * @param mapping            Contour length mapping
 * @param in_node            Source node
 * @param out_node           Target node
 *
 * @return String code using length index
 */
std::string PropagatorCode::generate_edge_code_with_mapping(
    std::map<std::pair<int, int>, std::string>& memory,
    std::vector<Block>& blocks,
    std::map<int, std::vector<int>>& adjacent_nodes,
    std::map<std::pair<int, int>, int>& edge_to_block_index,
    std::map<int, std::string>& chain_end_to_q_init,
    const ContourLengthMapping& mapping,
    int in_node, int out_node)
{
    std::vector<std::string> queue_sub_codes;
    std::string sub_code;

    // If it is already computed
    if (validation::contains(memory, std::make_pair(in_node, out_node)))
        return memory[std::make_pair(in_node, out_node)];

    // Explore neighbor nodes
    for(size_t i=0; i<adjacent_nodes[in_node].size(); i++)
    {
        if (adjacent_nodes[in_node][i] != out_node)
        {
            auto v_u_pair = std::make_pair(adjacent_nodes[in_node][i], in_node);
            if (validation::contains(memory, v_u_pair))
                sub_code = memory[v_u_pair];
            else
            {
                sub_code = generate_edge_code_with_mapping(
                    memory, blocks, adjacent_nodes,
                    edge_to_block_index, chain_end_to_q_init,
                    mapping,
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
        // Count each sub_code
        std::vector<std::string> unique_sub_codes;
        std::vector<int> counts;
        for(size_t i=0; i<queue_sub_codes.size(); i++)
        {
            if (i==0 || queue_sub_codes[i] != queue_sub_codes[i-1])
            {
                unique_sub_codes.push_back(queue_sub_codes[i]);
                counts.push_back(1);
            }
            else
            {
                counts[counts.size()-1] += 1;
            }
        }
        // Merge codes
        code += "(";
        for(size_t i=0; i<unique_sub_codes.size(); i++)
        {
            code += unique_sub_codes[i];
            if (counts[i] > 1)
            {
                code += ":" + std::to_string(counts[i]);
            }
        }
        code += ")";
    }
    // If in_node exists in chain_end_to_q_init
    else if (validation::contains(chain_end_to_q_init, in_node))
    {
        code = "{" + chain_end_to_q_init[in_node] + "}";
    }

    // Add monomer_type and length_index at the end of code
    int block_idx = edge_to_block_index[std::make_pair(in_node, out_node)];
    code += blocks[block_idx].monomer_type;
    // Use length index instead of n_segment
    int length_index = mapping.get_length_index(blocks[block_idx].contour_length);
    code += std::to_string(length_index);

    // Save the result in memory
    memory[std::make_pair(in_node, out_node)] = code;

    return code;
}

/**
 * @brief Extract propagator key by removing trailing segment count.
 *
 * Given a full code like "A10", returns "A" (the key without segment count).
 * For complex codes like "(A10B5)C8", returns "(A10B5)C".
 *
 * @param code Full propagator code with segment count
 * @return Key portion (code without trailing digits)
 */
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

/**
 * @brief Parse dependency information from a propagator key.
 *
 * Extracts the sub-propagators that must be computed before this one.
 * Parses the nested structure to find immediate dependencies.
 *
 * **Example:**
 *
 * Key "(A10B5:2)C" has dependencies:
 * - ("A", 10, 1) - one A propagator with 10 segments
 * - ("B", 5, 2) - two identical B propagators with 5 segments
 *
 * @param key Propagator key to parse
 * @return Vector of (sub_key, n_segment, n_repeated) tuples
 */
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

/**
 * @brief Remove outer monomer type from key, returning inner content.
 *
 * For keys like "(A10B5)C", returns "A10B5" (content inside parentheses).
 * For simple keys like "A", returns empty string.
 *
 * @param key Propagator key
 * @return Inner content without outer monomer type, or empty if no nesting
 */
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

/**
 * @brief Extract monomer type from the end of a propagator key.
 *
 * Returns the monomer type string at the outermost level of the key.
 * For "(A10B5)C", returns "C". For "A", returns "A".
 *
 * @param key Propagator key
 * @return Monomer type string
 */
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

/**
 * @brief Extract custom initial condition identifier from key.
 *
 * For keys starting with "{...}", extracts the identifier inside braces.
 * Used when chain ends have custom initial conditions (q_init).
 *
 * @param key Propagator key starting with "{"
 * @return Initial condition identifier
 * @throws Exception if key doesn't start with "{"
 */
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

/**
 * @brief Get nesting depth (height) of a propagator key.
 *
 * Counts the number of opening parentheses/brackets at the start.
 * Height 0 = chain end, Height 1+ = junction with merged branches.
 *
 * **Examples:**
 *
 * - "A" → height 0 (chain end)
 * - "(A10)B" → height 1 (one junction)
 * - "((A10)B5)C" → height 2 (nested junctions)
 *
 * @param key Propagator key
 * @return Nesting depth (0 for chain ends)
 */
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