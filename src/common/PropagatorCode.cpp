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

        // Code format is DKN (no ds_index in code)
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
 * @brief Extract propagator key from full code (legacy version without ds_index).
 *
 * Strips trailing length_index digits to get DK format.
 * For codes like "A3", returns "A".
 * For complex codes like "(A1B2)C3", returns "(A1B2)C".
 *
 * @param code Full propagator code (format: DKN)
 * @return Key in DK format (without ds_index)
 */
std::string PropagatorCode::get_key_from_code(std::string code)
{
    // Strip trailing digits (length_index) to get DK
    int pos = 0;
    for (int i = code.size() - 1; i >= 0; i--)
    {
        if (isalpha(code[i]))
        {
            pos = i + 1;
            break;
        }
    }
    return code.substr(0, pos);
}

/**
 * @brief Extract propagator key from full code with ds_index computation.
 *
 * Converts code format DKN to key format DK+M by:
 * 1. Extracting length_index N from the code
 * 2. Looking up ds_index M from the mapping
 * 3. Returning DK+M format
 *
 * For code "A3", returns "A+1" (if length_index 3 maps to ds_index 1).
 * For complex code "(A1B2)C3", returns "(A1B2)C+1".
 *
 * @param code Full propagator code (format: DKN)
 * @param mapping ContourLengthMapping for ds_index lookup
 * @return Key in DK+M format
 */
std::string PropagatorCode::get_key_from_code(std::string code, const ContourLengthMapping& mapping)
{
    // Find the position after the last alpha character (end of DK)
    int pos = 0;
    for (int i = code.size() - 1; i >= 0; i--)
    {
        if (isalpha(code[i]))
        {
            pos = i + 1;
            break;
        }
    }

    // Extract DK part and length_index
    std::string dk_part = code.substr(0, pos);
    int length_index = std::stoi(code.substr(pos));

    // Look up ds_index from length_index via mapping
    double contour_length = mapping.get_length_from_index(length_index);
    int ds_index = mapping.get_ds_index(contour_length);

    // Return DK+M format
    return dk_part + "+" + std::to_string(ds_index);
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
 * For aggregated keys (DKN format inside brackets):
 * Key "[(A)B3,(C)D2]E+1" has dependencies:
 * - ("(A)B", 3, 1) - propagator (A)B up to 3 segments
 * - ("(C)D", 2, 1) - propagator (C)D up to 2 segments
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
    // States: 1 = reading key, 2 = reading n_segment, 3 = reading n_repeated

    // First pass: check if key uses ; separator format (has ; at depth 1)
    // If so, use ; for ALL deps; otherwise use digit-based parsing
    bool use_semicolon_format = false;
    {
        std::stack<char> temp_stack;
        for(size_t i=0; i<key.size(); i++)
        {
            if(key[i] == '(' || key[i] == '[')
                temp_stack.push(key[i]);
            else if(key[i] == ')' || key[i] == ']')
                temp_stack.pop();
            else if(key[i] == ';' && temp_stack.size() == 1)
            {
                use_semicolon_format = true;
                break;
            }
        }
    }

    for(size_t i=0; i<key.size();i++)
    {
        if(stack_brace.size() == 1)
        {
            // For keys using ; separator format: look for ';' to find end of key
            if(state==1 && key[i]==';')
            {
                // Remove a comma at start
                if (key[pos_start] == ',')
                    pos_start += 1;

                sub_key = key.substr(pos_start, i-pos_start);
                state = 2;
                pos_start = i+1; // skip the ';'
            }
            // For non-aggregated keys: look for first digit after alpha (length_index)
            // Only use digit-based parsing if the key doesn't use ; format
            else if(state==1 && isdigit(key[i]) && !use_semicolon_format)
            {
                // Remove a comma
                if (key[pos_start] == ',')
                    pos_start += 1;

                sub_key = key.substr(pos_start, i-pos_start);

                state = 2;
                pos_start = i;
            }
            // It was reading n_segment (or length_index) and have found a ':'
            else if(state==2 && key[i]==':')
            {
                sub_n_segment = std::stoi(key.substr(pos_start, i-pos_start));

                state = 3;
                pos_start = i+1;
            }
            // It was reading n_segment (or length_index) and have found a non-digit
            else if(state==2 && !isdigit(key[i]))
            {
                sub_n_segment = std::stoi(key.substr(pos_start, i-pos_start));

                sub_deps.push_back(std::make_tuple(sub_key, sub_n_segment, 1));

                state = 1;
                pos_start = i;
            }
            // It was reading n_repeated and have found a non-digit
            else if(state==3 && !isdigit(key[i]))
            {
                sub_n_repeated = std::stoi(key.substr(pos_start, i-pos_start));

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
 * For key in DKM format "(A1B2)C0", returns "C".
 * For simple key "A0", returns "A".
 *
 * @param key Propagator key in DKM format
 * @return Monomer type string
 */
std::string PropagatorCode::get_monomer_type_from_key(std::string key)
{
    // First, strip the +M suffix if present (new format: DK+M)
    size_t plus_pos = key.rfind('+');
    if (plus_pos != std::string::npos)
        key = key.substr(0, plus_pos);

    // Find position after last closing bracket
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
    // Return the monomer type (everything after last closing bracket, or entire key)
    return key.substr(pos_start, key.size()-pos_start);
}

/**
 * @brief Extract ds_index from the end of a propagator key.
 *
 * For key in DK+M format "(A1B2)C+1", returns 1.
 * For simple key "A+2", returns 2.
 * Returns -1 if no ds_index is found (no '+' separator).
 * Note: ds_index uses 1-based indexing.
 *
 * @param key Propagator key in DK+M format
 * @return ds_index value (1-based), or -1 if not present
 */
int PropagatorCode::get_ds_index_from_key(std::string key)
{
    // Find the plus sign that separates DK from M
    size_t plus_pos = key.rfind('+');
    if (plus_pos == std::string::npos || plus_pos >= key.size() - 1)
        return -1;

    // Extract and parse the ds_index after '+'
    std::string ds_index_str = key.substr(plus_pos + 1);
    return std::stoi(ds_index_str);
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