/*----------------------------------------------------------
* This class constains functions for propagator code
*-----------------------------------------------------------*/

#ifndef PROPAGATOR_OPTIMIZATION_H_
#define PROPAGATOR_OPTIMIZATION_H_

#include <string>
#include <vector>
#include <map>
#include "Polymer.h"

// struct EssentialEdge{
//     int max_n_segment;                                    // the maximum segment number
//     std::string monomer_type;                             // monomer_type
//     std::vector<std::tuple<std::string, int, int>> deps;  // tuple <key, n_segment, n_repeated>
//     int height;                                           // height of propagator (height of tree data Structure)
// };
// struct EssentialBlock{
//     std::string monomer_type;  // monomer_type

//     // When the 'reduce_propagator_computation' is on, original block can be sliced to smaller block pieces.
//     // For example, suppose one block is composed of 5'A' monomers. -> -A-A-A-A-A-
//     // And this block is sliced into 3'A' monomers and 2'A' monomers -> -A-A-A-,  -A-A-
//     // For the first slice, n_segment_allocated, n_segment_offset, and n_segment_original are 3, 0, and 5, respectively.
//     // For the second slice, n_segment_allocated, n_segment_offset, and n_segment_original are 2, 3, and 5, respectively.
//     // If the 'reduce_propagator_computation' is off, original block is not sliced to smaller block pieces.
//     // In this case, n_segment_allocated, n_segment_offset, and n_segment_original are 5, 0, and 5, respectively.

//     int n_segment_allocated;
//     int n_segment_offset;      
//     int n_segment_original;
//     std::vector<std::tuple<int ,int>> v_u; // node pair <polymer id, v, u>
// };

// /* This stucture defines comparison function for branched key */
// struct ComparePropagatorKey
// {
//     bool operator()(const std::string& str1, const std::string& str2);
// };

// class Molecules
// {
// private:
//     std::string model_name; // "continuous": continuous standard Gaussian model
//                             // "discrete": discrete bead-spring model
                            
//     bool reduce_propagator_computation; // compute multiple propagators using property of linearity of the diffusion equation.

//     // distinct_polymers
//     std::vector<Polymer> distinct_polymers;

//     // set{key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u))}
//     std::map<std::tuple<int, std::string, std::string>, EssentialBlock> essential_blocks;

//     // dictionary{key:non-duplicated unique propagator_codes, value: EssentialEdge}
//     std::map<std::string, EssentialEdge, ComparePropagatorKey> essential_propagator_codes; 

//     // Get propagator code
//     // This method is implemented using memoization(top-down dynamic programming) approach.
//     std::pair<std::string, int> generate_propagator_code(
//         std::map<std::pair<int, int>, std::pair<std::string, int>>& memory,
//         std::vector<Block>& blocks,
//         std::map<int, std::vector<int>>& adjacent_nodes,
//         std::map<std::pair<int, int>, int>& edge_to_block_index,
//         std::map<int, std::string>& chain_end_to_q_init,
//         int in_node, int out_node);

//     // Add new key. if it already exists and 'new_n_segment' is larger than 'max_n_segment', update it.
//     void update_essential_propagator_code(std::map<std::string, EssentialEdge, ComparePropagatorKey>& essential_propagator_codes, std::string new_key, int new_n_segment);

//     // Superpose propagators
//     std::map<std::string, EssentialBlock> superpose_propagator_common             (std::map<std::string, EssentialBlock> remaining_keys, int minimum_n_segment);
//     std::map<std::string, EssentialBlock> superpose_propagator_of_continuous_chain(std::map<std::string, EssentialBlock> u_map);
//     std::map<std::string, EssentialBlock> superpose_propagator_of_discrete_chain  (std::map<std::string, EssentialBlock> u_map);

// public:

//     Molecules(std::string model_name, double ds, std::map<std::string, double> bond_lengths, bool reduce_propagator_computation);
//     ~Molecules() {};

//     std::string get_model_name() const;
//     double get_ds() const;
//     const std::map<std::string, double>& get_bond_lengths() const;
//     bool is_using_superposition() const;

//     // Add new polymer
//     // Mark some chain ends to set initial conditions of propagators when pseudo.compute_statistics() is invoked.
//     // For instance, if chain_end_to_q_init[a] is set to b, 
//     // Q_init[b] will be used as an initial condition of chain end 'a' in pseudo.compute_statistics().
//     void add_polymer(
//         double volume_fraction,
//         std::vector<BlockInput> block_inputs,
//         std::map<int, std::string> chain_end_to_q_init);

//     // Add new polymer
//     // All chain ends are free ends, e.g, q(r,0) = 1.
//     void add_polymer(
//         double volume_fraction,
//         std::vector<BlockInput> block_inputs)
//     {
//         add_polymer(volume_fraction, block_inputs, {});
//     }

//     // Get polymers
//     int get_n_polymer_types() const;
//     Polymer& get_polymer(const int p);

//     // Get information of essential propagators and blocks
//     int get_n_essential_propagator_codes() const;
//     std::map<std::string, EssentialEdge, ComparePropagatorKey>& get_essential_propagator_codes(); 
//     EssentialEdge& get_essential_propagator_code(std::string key);
//     std::map<std::tuple<int, std::string, std::string>, EssentialBlock>& get_essential_blocks(); 
//     EssentialBlock& get_essential_block(std::tuple<int, std::string, std::string> key);

//     // Get information from key
//     static std::vector<std::tuple<std::string, int, int>> get_deps_from_key(std::string key);
//     static std::string remove_monomer_type_from_key(std::string key);
//     static std::string get_monomer_type_from_key(std::string key);
//     static std::string get_q_input_idx_from_key(std::string key);
//     static int get_height_from_key(std::string key);

//     // Display
//     void display_propagators() const;
//     void display_blocks() const;
//     void display_sub_propagators() const;
// };
#endif
