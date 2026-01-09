/**
 * @file Polymer.cpp
 * @brief Implementation of Polymer class for chain architecture.
 *
 * Handles polymer topology as a graph structure where:
 * - Nodes: Junction points and chain ends
 * - Edges: Polymer blocks (connecting nodes v and u)
 *
 * **Graph Validation:**
 *
 * The constructor validates that the polymer graph is:
 * 1. Acyclic (no closed loops)
 * 2. Connected (all nodes reachable from first block)
 * 3. Has unique edges (no duplicate blocks between same nodes)
 *
 * Uses depth-first search to detect cycles and isolated nodes.
 *
 * **Propagator Key Assignment:**
 *
 * Each directed edge (v→u) is assigned a unique propagator key
 * using PropagatorCode::generate_codes() for efficient computation.
 *
 * @see PropagatorCode for key generation algorithm
 * @see Molecules for polymer container
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cctype>
#include <cmath>
#include <climits>
#include <numbers>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "Polymer.h"
#include "PropagatorCode.h"
#include "ContourLengthMapping.h"
#include "Exception.h"
#include "ValidationUtils.h"

/**
 * @brief Construct polymer from block specification.
 *
 * Builds the polymer graph, validates acyclicity and connectivity,
 * then generates optimized propagator codes.
 *
 * @param ds                  Contour step size
 * @param bond_lengths        Segment lengths by monomer type
 * @param volume_fraction     Polymer volume fraction in blend
 * @param block_inputs        Block definitions [monomer, length, v, u]
 * @param chain_end_to_q_init Custom initial conditions at chain ends
 *
 * @throws Exception if graph has cycles, is disconnected, or has invalid blocks
 */
Polymer::Polymer(
    double ds, std::map<std::string, double> bond_lengths, 
    double volume_fraction, std::vector<BlockInput> block_inputs,
    std::map<int, std::string> chain_end_to_q_init)
{

    // Check the name of monomer_type. Only alphabets and underscore are allowed.
    for(const auto& item : bond_lengths)
    {
        if (!std::all_of(item.first.begin(), item.first.end(), [](unsigned char c){ return std::isalpha(c) || c=='_' ; }))
        {
            throw_with_line_number("\"" + item.first + "\" is an invalid monomer_type name. Only alphabets and underscore(_) are allowed.")
        }
        validation::require_positive(item.second, "bond_lengths[\"" + item.first + "\"]");
    }

    // Check block lengths, segments, types
    for(size_t i=0; i<block_inputs.size(); i++)
    {
        validation::require_positive(block_inputs[i].contour_length,
            "block_inputs[" + std::to_string(i) + "].contour_length");
        // Note: contour_length/ds no longer needs to be exactly an integer.
        // The ContourLengthMapping class handles computing n_segment = round(contour_length/ds)
        // and the corresponding local ds = contour_length / n_segment.
        validation::require_string_key(bond_lengths, block_inputs[i].monomer_type, "bond_lengths");
    }

    // chain_end_to_q_init
    this->chain_end_to_q_init = chain_end_to_q_init;

    // Save variables
    try
    {
        this->volume_fraction = volume_fraction;
        for(size_t i=0; i<block_inputs.size(); i++){
            // Robust rounding that handles floating-point errors near 0.5 boundaries.
            // If the ratio is very close to a half-integer (e.g., 3.4999999999 or 3.5000000001),
            // snap it to the nearest 0.5 before rounding to ensure consistent results.
            double ratio = block_inputs[i].contour_length / ds;
            double rounded_to_half = std::round(ratio * 2.0) / 2.0;
            if (std::abs(ratio - rounded_to_half) < 1e-9)
            {
                ratio = rounded_to_half;
            }
            int n_segment = static_cast<int>(std::lround(ratio));
            if (n_segment < 1)
            {
                n_segment = 1;
            }

            blocks.push_back({
                block_inputs[i].monomer_type,                           // monomer_type
                n_segment,                                              // n_segment
                block_inputs[i].contour_length,                         // contour_length
                block_inputs[i].v,                                      // starting node
                block_inputs[i].u});                                    // ending node
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }

    // Compute alpha, sum of relative contour lengths
    double alpha{0.0};
    for(size_t i=0; i<blocks.size(); i++){
        alpha += blocks[i].contour_length;
    }
    this->alpha = alpha;

    // Construct starting vertices 'v', ending vertices 'u', 
    std::vector<int> v;
    std::vector<int> u;
    for(size_t i=0; i<block_inputs.size(); i++)
    {
        v.push_back(block_inputs[i].v);
        u.push_back(block_inputs[i].u);
    }

    // Construct adjacent_nodes
    for(size_t i=0; i<v.size(); i++)
    {
        adjacent_nodes[v[i]].push_back(u[i]);
        adjacent_nodes[u[i]].push_back(v[i]);
        // V and u must be a non-negative integer for depth first search
        validation::require_non_negative(v[i], "v[" + std::to_string(i) + "]");
        validation::require_non_negative(u[i], "u[" + std::to_string(i) + "]");
        if( v[i] == u[i] )
        {
            throw_with_line_number("v[" + std::to_string(i) + "] and u[" + std::to_string(i) + "] must be different integers.")
        }
    }

    // // Print adjacent_nodes
    // for(const auto& node : adjacent_nodes){
    //     std::cout << node.first << ": [";
    //     for(int i=0; i<node.second.size()-1; i++)
    //         std::cout << node.second[i] << ",";
    //     std::cout << node.second[node.second.size()-1] << "]" << std::endl;
    // }

    // Construct edge nodes
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

    // Detect a cycle and isolated nodes in the block copolymer graph using depth first search
    std::map<int, bool> is_visited;
    for (size_t i = 0; i < v.size(); i++)
        is_visited[v[i]] = false;
    for (size_t i = 0; i < u.size(); i++)
        is_visited[u[i]] = false;

    std::stack<std::pair<int,int>> connected_nodes;
    // Starting node is v[0]
    connected_nodes.push(std::make_pair(v[0],-1));
    // Perform depth first search
    while (!connected_nodes.empty())
    {
        // Get one item and visit
        int cur = connected_nodes.top().first;
        int parent = connected_nodes.top().second;
        is_visited[cur] = true;

        // Remove the item
        connected_nodes.pop();

        // Add adjacent_nodes at stack 
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

    // Collect isolated nodes
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

    // Print isolated nodes
    std::vector<int> isolated_nodes(isolated_nodes_set.begin(), isolated_nodes_set.end());
    if (isolated_nodes.size() > 0)
    {
        std::string error_message = "There is no route from node " + std::to_string(v[0]) + " to nodes: "
            + std::to_string(isolated_nodes[0]);
        for (size_t i=1; i<isolated_nodes.size(); i++)
            error_message += ", " + std::to_string(isolated_nodes[i]);
        throw_with_line_number(error_message + ".");
    }

    // Generate propagator codes and assign each of them to each of propagator
    auto propagator_codes = PropagatorCode::generate_codes(*this, chain_end_to_q_init);
    for(size_t i=0; i<propagator_codes.size(); i++)
    {
        int v = std::get<0>(propagator_codes[i]);
        int u = std::get<1>(propagator_codes[i]);
        std::string propagator_key = PropagatorCode::get_key_from_code(std::get<2>(propagator_codes[i]));
        this->set_propagator_key(propagator_key, v, u);
    }

}
int Polymer::get_n_blocks() const
{
    return blocks.size();
}
int Polymer::get_n_segment(const int idx) const
{
    return blocks[idx].n_segment;
}
int Polymer::get_n_segment_total() const
{
    int total_n{0};
    for(size_t i=0; i<blocks.size(); i++)
        total_n += blocks[i].n_segment;
    return total_n;
}
double Polymer::get_alpha() const
{
    return alpha;
}
double Polymer::get_volume_fraction() const
{
    return volume_fraction;
}
int Polymer::get_block_index_from_edge(const int v, const int u) const
{
    validation::require_key(edge_to_block_index, std::make_pair(v,u),
        "edge_to_block_index for (" + std::to_string(v) + ", " + std::to_string(u) + ")");
    return edge_to_block_index.at(std::make_pair(v, u));
}
struct Block& Polymer::get_block(const int v, const int u)
{
    validation::require_key(edge_to_block_index, std::make_pair(v,u),
        "edge_to_block_index for (" + std::to_string(v) + ", " + std::to_string(u) + ")");
    return blocks[edge_to_block_index[std::make_pair(v, u)]];
}
std::vector<Block>& Polymer::get_blocks()
{
    return blocks;
}
std::map<int, std::vector<int>>& Polymer::get_adjacent_nodes()
{
    return adjacent_nodes;
}
std::map<std::pair<int, int>, int>& Polymer::get_block_indexes()
{
    return edge_to_block_index;
}
void Polymer::set_propagator_key(const std::string code, const int v, const int u)
{
    edge_to_propagator_key[std::make_pair(v, u)] = code;
}
std::string Polymer::get_propagator_key(const int v, const int u) const {
    validation::require_key(edge_to_propagator_key, std::make_pair(v,u),
        "edge_to_propagator_key for (" + std::to_string(v) + ", " + std::to_string(u) + ")");
    return edge_to_propagator_key.at(std::make_pair(v,u));
}

std::map<int, std::string>& Polymer::get_chain_end_to_q_init()
{
    return chain_end_to_q_init;
}

/**
 * @brief Regenerate propagator keys using contour length mapping.
 *
 * Replaces existing keys with new ones that use length indices.
 *
 * @param mapping Contour length mapping (must be finalized)
 */
void Polymer::regenerate_propagator_keys(const ContourLengthMapping& mapping)
{
    // Clear existing keys
    edge_to_propagator_key.clear();

    // Generate new keys using the mapping
    // Code format: DKN, Key format: DK+M
    auto propagator_codes = PropagatorCode::generate_codes_with_mapping(*this, chain_end_to_q_init, mapping);
    for(size_t i=0; i<propagator_codes.size(); i++)
    {
        int v = std::get<0>(propagator_codes[i]);
        int u = std::get<1>(propagator_codes[i]);
        std::string propagator_key = PropagatorCode::get_key_from_code(std::get<2>(propagator_codes[i]), mapping);
        this->set_propagator_key(propagator_key, v, u);
    }
}

/**
 * @brief Helper function to format edge label with monomer type and contour length.
 */
static std::string format_edge(const Block& block)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << block.monomer_type << "[" << block.contour_length << "]";
    return oss.str();
}

/**
 * @brief Recursive helper to print tree structure.
 *
 * @param node Current node
 * @param parent Parent node (-1 for root)
 * @param prefix Current indentation prefix
 * @param is_last Whether this is the last child of parent
 */
void print_tree_recursive(
    int node,
    int parent,
    const std::string& prefix,
    [[maybe_unused]] bool is_last,
    const std::map<int, std::vector<int>>& adj_nodes,
    const std::map<std::pair<int, int>, int>& edge_to_idx,
    const std::vector<Block>& all_blocks)
{
    // Get children (neighbors except parent)
    std::vector<int> children;
    auto it = adj_nodes.find(node);
    if (it != adj_nodes.end())
    {
        for (int neighbor : it->second)
        {
            if (neighbor != parent)
            {
                children.push_back(neighbor);
            }
        }
    }
    std::sort(children.begin(), children.end());

    // Print each child
    for (size_t i = 0; i < children.size(); i++)
    {
        int child = children[i];
        bool child_is_last = (i == children.size() - 1);

        // Get block info for this edge
        auto edge_it = edge_to_idx.find(std::make_pair(node, child));
        if (edge_it == edge_to_idx.end())
        {
            edge_it = edge_to_idx.find(std::make_pair(child, node));
        }
        const Block& block = all_blocks[edge_it->second];
        std::string edge_label = format_edge(block);

        // Print edge and child node
        std::cout << prefix;
        std::cout << (child_is_last ? " \\--" : " +--");
        std::cout << edge_label << "--(" << child << ")" << std::endl;

        // Recurse with updated prefix
        std::string new_prefix = prefix + (child_is_last ? "    " : " |  ");
        // Add spacing for edge label width
        new_prefix += std::string(edge_label.length(), ' ');

        print_tree_recursive(child, node, new_prefix, child_is_last,
                            adj_nodes, edge_to_idx, all_blocks);
    }
}

void Polymer::print_architecture_diagram() const
{
    if (blocks.empty())
    {
        std::cout << "(empty polymer)" << std::endl;
        return;
    }

    // Find root: first vertex with degree 1, or smallest vertex if all have degree > 1
    int root = -1;
    int min_vertex = INT_MAX;

    for (const auto& pair : adjacent_nodes)
    {
        int v = pair.first;
        min_vertex = std::min(min_vertex, v);

        if (pair.second.size() == 1 && root == -1)
        {
            root = v;
        }
    }
    if (root == -1)
    {
        root = min_vertex;
    }

    // Check if linear chain (all vertices have degree <= 2)
    bool is_linear = true;
    for (const auto& pair : adjacent_nodes)
    {
        if (pair.second.size() > 2)
        {
            is_linear = false;
            break;
        }
    }

    if (is_linear && blocks.size() <= 10)
    {
        // Print linear chain in single line
        // Traverse from root following the chain
        std::set<int> visited;
        int current = root;
        std::cout << "(" << current << ")";
        visited.insert(current);

        while (true)
        {
            // Find unvisited neighbor
            int next = -1;
            auto it = adjacent_nodes.find(current);
            if (it != adjacent_nodes.end())
            {
                for (int neighbor : it->second)
                {
                    if (visited.find(neighbor) == visited.end())
                    {
                        next = neighbor;
                        break;
                    }
                }
            }

            if (next == -1)
            {
                break;
            }

            // Get block info
            auto edge_it = edge_to_block_index.find(std::make_pair(current, next));
            if (edge_it == edge_to_block_index.end())
            {
                edge_it = edge_to_block_index.find(std::make_pair(next, current));
            }
            const Block& block = blocks[edge_it->second];

            std::cout << "--" << format_edge(block) << "--(" << next << ")";
            visited.insert(next);
            current = next;
        }
        std::cout << std::endl;
    }
    else
    {
        // Print as tree
        std::cout << "(" << root << ")" << std::endl;
        print_tree_recursive(root, -1, "", true,
                            adjacent_nodes, edge_to_block_index, blocks);
    }
}

void Polymer::display_architecture() const
{
    std::cout << "=== Polymer ===" << std::endl;
    print_architecture_diagram();
    std::cout << "Legend: (n)=vertex, Type[length]=block" << std::endl;
}