/*----------------------------------------------------------
* This class defines analyzer of propagator to optimize the propagator computation
*-----------------------------------------------------------*/

#ifndef PROPAGATOR_ANALYZER_H_
#define PROPAGATOR_ANALYZER_H_

#include <string>
#include <vector>
#include <map>

#include "Molecules.h"
#include "Polymer.h"

struct ComputationEdge{
    int max_n_segment;                                    // the maximum segment number
    std::string monomer_type;                             // monomer_type
    std::vector<std::tuple<std::string, int, int>> deps;  // tuple <key, n_segment, n_repeated>
    int height;                                           // height of propagator (height of tree data Structure)
};
struct ComputationBlock{
    std::string monomer_type;  // monomer_type

    // When the 'aggregate_propagator_computation' is on, one block can be sliced to smaller block pieces.
    // For example, suppose the block is composed of 5'A' monomers. -> -A-A-A-A-A-
    // , and this block is sliced into 3'A' monomers and 2'A' monomers -> -A-A-A-,  -A-A-
    // For the first slice, n_segment_right, and n_segment_left are 2, and 5, respectively.
    // For the second slice, n_segment_right, and n_segment_left are 3, and 3, respectively.
    // If the 'aggregate_propagator_computation' is off, original block is not sliced to smaller block pieces.
    // In this case, n_segment_right, and n_segment_left are 5, and 5, respectively.
    int n_segment_left;
    int n_segment_right;
    int n_repeated;
    std::vector<std::tuple<int ,int>> v_u; // node pair <polymer id, v, u>
};

/* This stucture defines comparison function for branched key */
struct ComparePropagatorKey
{
    bool operator()(const std::string& str1, const std::string& str2);
};

class PropagatorAnalyzer
{
private:
    std::string model_name; // "continuous": continuous standard Gaussian model
                            // "discrete": discrete bead-spring model
    bool aggregate_propagator_computation; // compute multiple propagators using property of linearity of the diffusion equation.

    // set{key: (polymer id, key_left, key_right) (assert(key_left <= key_right))}
    std::map<std::tuple<int, std::string, std::string>, ComputationBlock> computation_blocks;

    // dictionary{key:non-duplicated unique propagator_codes, value: ComputationEdge}
    std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagator_codes; 

    // Total segment number
    std::vector<int> total_segment_numbers;

    // Substitute right keys of lower left key with aggregated keys
    void substitute_right_keys(
        Polymer& pc, 
        std::map<std::tuple<int, int>, std::string>& v_u_to_right_key,
        std::map<std::string, std::map<std::string, ComputationBlock>> & computation_blocks_new_polymer,
        std::map<std::string, std::vector<std::string>>& aggregated_blocks,
        std::string left_key);

    // Add new key. if it already exists and 'new_n_segment' is larger than 'max_n_segment', update it.
    void update_computation_propagator_map(std::map<std::string, ComputationEdge, ComparePropagatorKey>& computation_propagator_codes, std::string new_key, int new_n_segment);

public:
    PropagatorAnalyzer(Molecules* molecules, bool aggregate_propagator_computation);
    // ~PropagatorAnalyzer() {};

    // Add new polymers
    void add_polymer(Polymer& pc, int polymer_count);

    // Aggregate propagators
    static std::map<std::string, ComputationBlock> aggregate_propagator_common          (std::map<std::string, ComputationBlock> remaining_keys, int minimum_n_segment);
    static std::map<std::string, ComputationBlock> aggregate_propagator_continuous_chain(std::map<std::string, ComputationBlock> u_map);
    static std::map<std::string, ComputationBlock> aggregate_propagator_discrete_chain  (std::map<std::string, ComputationBlock> u_map);

    // Get information of computation propagators and blocks
    bool is_aggregated() const;
    int get_n_computation_propagator_codes() const;
    std::map<std::string, ComputationEdge, ComparePropagatorKey>& get_computation_propagator_codes(); 
    ComputationEdge& get_computation_propagator_code(std::string key);
    std::map<std::tuple<int, std::string, std::string>, ComputationBlock>& get_computation_blocks(); 
    ComputationBlock& get_computation_block(std::tuple<int, std::string, std::string> key);

    // Display
    void display_propagators() const;
    void display_blocks() const;
    void display_sub_propagators() const;
};
#endif
